use crate::transformer::CommandsTransformer;
use std::borrow::Borrow;
use syn::parse::{Parse, ParseStream};

struct CommandsTransformState {
    current_import_id: usize,
    import_bindings: proc_macro2::TokenStream,
    import_drops: proc_macro2::TokenStream,
    import_retained_states: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,

    current_await_index: usize,
    await_bindings: proc_macro2::TokenStream,
    await_drops: proc_macro2::TokenStream,
    await_retained_states: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
}
impl Default for CommandsTransformState {
    fn default() -> Self {
        use proc_macro2::TokenStream;
        use syn::punctuated::Punctuated;
        Self {
            current_import_id: 0,
            import_bindings: TokenStream::new(),
            import_drops: TokenStream::new(),
            import_retained_states: Punctuated::new(),

            current_await_index: 0,
            await_bindings: TokenStream::new(),
            await_drops: TokenStream::new(),
            await_retained_states: Punctuated::new(),
        }
    }
}
impl CommandsTransformer for CommandsTransformState {
    fn import(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        is_image: bool,
    ) -> proc_macro2::TokenStream {
        let global_res_variable_name =
            quote::format_ident!("__future_res_{}", self.current_import_id);
        self.current_import_id += 1;
        let output_tokens = if is_image {
            quote::quote! {{
                __fut_global_ctx.add_image(&mut #global_res_variable_name)
            }}
        } else {
            quote::quote! {{
                __fut_global_ctx.add_res(&mut #global_res_variable_name)
            }}
        };
        self.import_bindings.extend(quote::quote! {
            let mut #global_res_variable_name = #input_tokens;
        });
        self.import_drops.extend(quote::quote! {
            drop(#global_res_variable_name);
        });
        self.import_retained_states
            .push(syn::Expr::Verbatim(quote::quote! {
                #global_res_variable_name
            }));
        output_tokens
    }
    fn async_transform(&mut self, input: &syn::ExprAwait) -> syn::Expr {
        let global_future_variable_name =
            quote::format_ident!("__future_{}", self.current_await_index);
        self.current_await_index += 1;

        self.await_bindings.extend(quote::quote! {
            let mut #global_future_variable_name;
        });
        self.await_drops.extend(quote::quote! {
            drop(#global_future_variable_name);
        });
        self.await_retained_states
            .push(syn::Expr::Verbatim(quote::quote! {
                #global_future_variable_name
            }));

        let base = input.base.clone();

        // For each future, we first call init on it. This is a no-op for most futures, but for
        // (nested) block futures, this is going to move the future to its first yield point.
        // At that point it should yield the context of the first future.
        let tokens = quote::quote! {
            {
                let mut fut = #base;
                let mut fut_pinned = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
                fut_pinned.as_mut().init(__fut_global_ctx);
                let mut ctx = Default::default();
                fut_pinned.as_mut().context(&mut ctx);
                yield ctx;
                loop {
                    use ::std::task::Poll::*;
                    match fut_pinned.as_mut().record(__fut_global_ctx) {
                        Ready((output, retained_state)) => {
                            #global_future_variable_name = retained_state;
                            break output
                        },
                        Yielded => {
                            let mut ctx = Default::default();
                            fut_pinned.as_mut().context(&mut ctx);
                            yield ctx;
                        },
                    };
                }
            }
        };
        syn::parse2(tokens).unwrap()
    }
    fn macro_transform(&mut self, mac: &syn::ExprMacro) -> syn::Expr {
        let path = &mac.mac.path;
        if path.segments.len() != 1 {
            return syn::Expr::Macro(mac.clone());
        }
        match path.segments[0].ident.to_string().as_str() {
            "import" => syn::Expr::Verbatim(self.import(&mac.mac.tokens, false)),
            "import_image" => syn::Expr::Verbatim(self.import(&mac.mac.tokens, true)),
            "fork" => syn::Expr::Verbatim(self.fork_transform(&mac.mac.tokens)),
            _ => syn::Expr::Macro(mac.clone()),
        }
    }
    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr> {
        // Transform each return statement into a yield, drops, and return.
        // We use RefCell on awaited_future_drops and import_drops so that they can be read while being modified.
        // This ensures that we won't drop uninitialized values.'
        // The executor will stop the execution as soon as it reaches the first `yield Complete` statement.
        // Drops are written between the yield and return, so these values are retained inside the generator
        // until the generator itself was dropped.
        // We drop the generator only after semaphore was signaled from within the queue.
        let returned_item = ret.expr.as_ref().map(|a| *a.clone()).unwrap_or(syn::Expr::Verbatim(quote::quote!(())));
        let awaited_future_retained_states = &*self.await_retained_states.borrow();
        let import_retained_states = &self.import_retained_states;
        let token_stream = quote::quote!(
            {
                return (#returned_item, ((#awaited_future_retained_states), (#import_retained_states)));
            }
        );
        let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
        Some(syn::Expr::Block(block))
    }
}
impl CommandsTransformState {
    fn fork_transform(&mut self, input: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let ForkInput {
            forked_future,
            number_of_forks,
            comma: _,
            scope,
        } = match syn::parse2::<ForkInput>(input.clone()) {
            Ok(input) => input,
            Err(err) => return err.to_compile_error(),
        };
        let number_of_forks = number_of_forks.map(|(_, number)| number).unwrap_or(2);
        let ret = syn::Expr::Tuple(syn::ExprTuple {
            attrs: Vec::new(),
            paren_token: Default::default(),
            elems: (0..number_of_forks)
                .map(|i| {
                    syn::Expr::Verbatim(quote::quote! {
                        GPUCommandForked::new(&forked_future_inner, #i)
                    })
                })
                .collect(),
        });
        let scope = syn::Block {
            brace_token: scope.brace_token.clone(),
            stmts: scope
                .stmts
                .iter()
                .map(|stmt| self.transform_stmt(stmt))
                .collect(),
        };
        quote::quote! {{
            let mut forked_future = ::async_ash::future::GPUCommandForkedStateInner::Some(#forked_future);
            let mut pinned = unsafe{std::pin::Pin::new_unchecked(&mut forked_future)};
            pinned.as_mut().unwrap_pinned().init(__fut_global_ctx);
            let forked_future_inner = GPUCommandForkedInner::<_, #number_of_forks>::new(pinned);
            let #forked_future = #ret;
            #scope
        }}
    }
}

struct ForkInput {
    forked_future: syn::Expr,
    number_of_forks: Option<(syn::Token![,], usize)>,
    comma: syn::Token![,],
    scope: syn::Block,
}
impl Parse for ForkInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let forked_future = input.parse()?;
        let comma: syn::Token![,] = input.parse()?;
        let number_of_forks = {
            let number: Option<syn::LitInt> = input.parse().ok();
            let number = number.and_then(|a| a.base10_parse::<usize>().ok());
            number
        };
        if let Some(number_of_forks) = number_of_forks {
            Ok(ForkInput {
                forked_future,
                number_of_forks: Some((comma, number_of_forks)),
                comma: input.parse()?,
                scope: input.parse()?,
            })
        } else {
            Ok(ForkInput {
                forked_future,
                number_of_forks: None,
                comma,
                scope: input.parse()?,
            })
        }
    }
}

pub fn proc_macro_commands(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<crate::ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut state = CommandsTransformState::default();

    let mut inner_closure_stmts: Vec<_> = input
        .stmts
        .iter()
        .map(|stmt| state.transform_stmt(stmt))
        .collect();

    // Transform the final return
    if let Some(last) = inner_closure_stmts.last_mut() {
        let import_retained_states = &*state.import_retained_states.borrow();
        let awaited_future_retained_states = &*state.await_retained_states.borrow();

        if let syn::Stmt::Expr(expr) = last {
            let token_stream = quote::quote!(
                {
                    return (#expr, ((#awaited_future_retained_states), (#import_retained_states)));
                }
            );
            let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
            *expr = syn::Expr::Block(block);
        } else {
            let token_stream = quote::quote!(
                {
                    return ((), ((#awaited_future_retained_states), (#import_retained_states)));
                }
            );
            let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
            inner_closure_stmts.push(syn::Stmt::Expr(syn::Expr::Block(block)))
        }
    }

    let import_bindings = state.import_bindings;
    let awaited_future_bindings = state.await_bindings;
    // todo: __fut_global_ctx dereference on-site.
    quote::quote! {
        async_ash::future::GPUCommandBlock::new(static |__fut_global_ctx: *mut ::async_ash::future::GlobalContext| {
            let __fut_global_ctx: &mut ::async_ash::future::GlobalContext = unsafe{&mut *__fut_global_ctx};
            #import_bindings
            #awaited_future_bindings
            #(#inner_closure_stmts)*
        })
    }
}
